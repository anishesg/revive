package com.revive.worker

import android.content.Intent
import android.os.Bundle
import android.os.Environment
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.io.File

val ReviveGreen = Color(0xFF00FF88)
val ReviveBlue = Color(0xFF4A90D9)
val ReviveBg = Color(0xFF0A0A0A)
val ReviveCard = Color(0xFF111111)
val ReviveBorder = Color(0xFF222222)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ReviveTheme {
                ReviveApp(
                    onStartWorker = { role, modelPath, port ->
                        startWorkerService(role, modelPath, port)
                    },
                    onStopWorker = { stopWorkerService() },
                    modelsDir = getModelsDir(),
                )
            }
        }
    }

    private fun getModelsDir(): File {
        val dir = File(getExternalFilesDir(null), "models")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    private fun startWorkerService(role: String, modelPath: String, port: Int) {
        val intent = Intent(this, WorkerService::class.java).apply {
            putExtra(WorkerService.EXTRA_MODEL_PATH, modelPath)
            putExtra(WorkerService.EXTRA_ROLE, role)
            putExtra(WorkerService.EXTRA_PORT, port)
        }
        startForegroundService(intent)
    }

    private fun stopWorkerService() {
        stopService(Intent(this, WorkerService::class.java))
    }
}

@Composable
fun ReviveTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            background = ReviveBg,
            surface = ReviveCard,
            primary = ReviveGreen,
            onBackground = Color.White,
            onSurface = Color.White,
        ),
        content = content,
    )
}

@Composable
fun ReviveApp(onStartWorker: (String, String, Int) -> Unit, onStopWorker: () -> Unit, modelsDir: File) {
    var isRunning by remember { mutableStateOf(false) }
    var selectedRole by remember { mutableStateOf("drafter") }
    var selectedModel by remember { mutableStateOf<File?>(null) }
    var port by remember { mutableStateOf("8080") }

    val models = remember(modelsDir) {
        modelsDir.listFiles { f -> f.name.endsWith(".gguf") }?.toList() ?: emptyList()
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(ReviveBg)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth()) {
                Text(
                    "R E V I V E",
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Black,
                    fontFamily = FontFamily.Monospace,
                    color = ReviveGreen,
                )
                Text(
                    "Android Worker Node",
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace,
                    color = Color.Gray,
                )
            }
        }

        item {
            SectionHeader("MODEL")
            if (models.isEmpty()) {
                Text(
                    "No models found. Place .gguf files in:\n${modelsDir.absolutePath}",
                    color = Color(0xFFFF6B6B),
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(8.dp),
                )
            }
        }

        items(models) { model ->
            val isSelected = selectedModel == model
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(if (isSelected) ReviveCard else Color.Transparent)
                    .border(1.dp, if (isSelected) ReviveGreen.copy(alpha = 0.4f) else ReviveBorder, RoundedCornerShape(8.dp))
                    .clickable { selectedModel = model }
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(if (isSelected) ReviveGreen else Color.Gray),
                )
                Spacer(Modifier.width(12.dp))
                Column {
                    Text(model.name, color = Color.White, fontSize = 13.sp, fontFamily = FontFamily.Monospace)
                    Text(
                        "${model.length() / (1024 * 1024)}MB",
                        color = Color.Gray, fontSize = 11.sp, fontFamily = FontFamily.Monospace,
                    )
                }
            }
        }

        item {
            SectionHeader("ROLE")
            FlowRow(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AgentRoles.allRoles.forEach { role ->
                    val isSelected = selectedRole == role
                    Text(
                        role,
                        modifier = Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(if (isSelected) ReviveGreen.copy(alpha = 0.2f) else Color.Transparent)
                            .border(1.dp, if (isSelected) ReviveGreen else ReviveBorder, RoundedCornerShape(6.dp))
                            .clickable { selectedRole = role }
                            .padding(horizontal = 12.dp, vertical = 8.dp),
                        color = if (isSelected) ReviveGreen else Color.Gray,
                        fontSize = 12.sp,
                        fontFamily = FontFamily.Monospace,
                    )
                }
            }
        }

        item {
            SectionHeader("PORT")
            OutlinedTextField(
                value = port,
                onValueChange = { port = it.filter { c -> c.isDigit() } },
                modifier = Modifier.fillMaxWidth(),
                textStyle = LocalTextStyle.current.copy(fontFamily = FontFamily.Monospace, color = Color.White),
                singleLine = true,
            )
        }

        item {
            Spacer(Modifier.height(8.dp))
            Button(
                onClick = {
                    if (isRunning) {
                        onStopWorker()
                        isRunning = false
                    } else {
                        selectedModel?.let { model ->
                            onStartWorker(selectedRole, model.absolutePath, port.toIntOrNull() ?: 8080)
                            isRunning = true
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth().height(56.dp),
                enabled = selectedModel != null || isRunning,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isRunning) Color(0xFFD0021B) else ReviveGreen,
                    contentColor = if (isRunning) Color.White else Color.Black,
                ),
                shape = RoundedCornerShape(12.dp),
            ) {
                Text(
                    if (isRunning) "STOP WORKER" else "START WORKER",
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 16.sp,
                )
            }
        }

        if (isRunning) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = ReviveCard),
                    shape = RoundedCornerShape(8.dp),
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.size(8.dp).clip(CircleShape).background(ReviveGreen))
                            Spacer(Modifier.width(8.dp))
                            Text("ACTIVE", color = ReviveGreen, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, fontSize = 12.sp)
                        }
                        Spacer(Modifier.height(8.dp))
                        Text("Role: $selectedRole", color = Color.Gray, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                        Text("Port: $port", color = Color.Gray, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                        Text("Model: ${selectedModel?.name}", color = Color.Gray, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                        Spacer(Modifier.height(8.dp))
                        Text("The coordinator will discover this device automatically via mDNS.", color = Color(0xFF666666), fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                    }
                }
            }
        }
    }
}

@Composable
fun SectionHeader(text: String) {
    Text(
        text,
        color = ReviveBlue,
        fontSize = 11.sp,
        fontWeight = FontWeight.Bold,
        fontFamily = FontFamily.Monospace,
        letterSpacing = 2.sp,
        modifier = Modifier.padding(bottom = 8.dp),
    )
}

@Composable
fun FlowRow(
    modifier: Modifier = Modifier,
    horizontalArrangement: Arrangement.Horizontal = Arrangement.Start,
    content: @Composable () -> Unit,
) {
    Row(modifier = modifier, horizontalArrangement = horizontalArrangement) {
        content()
    }
}
